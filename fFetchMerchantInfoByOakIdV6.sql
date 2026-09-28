/****** Object:  UserDefinedFunction [dbo].[fFetchMerchantInfoByOakIdV6]    Script Date: 9/28/2026 12:34:25 PM ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

SELECT * from [dbo].[fFetchMerchantInfoByOakIdV6](1129451, '77565416551', '451550')

CREATE     FUNCTION [dbo].[fFetchMerchantInfoByOakIdV6](@OakCardId INT, @CardAcceptorId varchar(100) = NULL, @AcquiringInstitutionIdCode varchar(16) = NULL)
RETURNS @ResTable TABLE (
	IsCardActive BIT,
	IsATAEnabled BIT,
	AvailableToAuthorize FLOAT,
	IsVCCDecisioning BIT,
	WebhookEndpoint VARCHAR(200),
	WebhookUserName VARCHAR(50),
	WebhookPassword VARCHAR(50),
	TimeoutHandling VARCHAR(50),
	VccGuid VARCHAR(50),
	Id TINYINT,
	IsAVSEnabled BIT,
	EnforceCVVCheck BIT,
	AllowPartialAuthorizations BIT,
	IsBlackListed BIT 
)
AS
BEGIN
	DECLARE @AvailableToAuthorize FLOAT = 0;
	DECLARE @merchantGuid UNIQUEIDENTIFIER;
	DECLARE @IdIncomingTransactionCode INT = 0;
	DECLARE @IdMerchant INT = 0;
	DECLARE @IdMasterMerchant INT = 0;
	DECLARE @VccGuid VARCHAR(50) = NULL;
	DECLARE @IsATAEnabled BIT = 0;
	DECLARE @IsCardActive BIT = 0;
	DECLARE @IsAVSEnabled BIT = 0;
	DECLARE @EnforceCVVCheck BIT = 0;
	DECLARE @AllowPartialAuthorizations BIT = 0;   
	DECLARE @IsVCCDecisioning BIT;
	DECLARE @WebhookEndpoint VARCHAR(200), @WebhookUserName VARCHAR(50), @WebhookPassword VARCHAR(50), @TimeoutHandling VARCHAR(50);
	DECLARE @ParentMerchantGuid UNIQUEIDENTIFIER;
	DECLARE @UseParentFunding BIT;
	DECLARE @IsBlackListed BIT = 0;
	SELECT
		@VccGuid = Guid,
		@IdIncomingTransactionCode = IdIncomingTransactionCode,
		@IsCardActive = CAST(CASE WHEN IdStatus = 400 THEN 1 ELSE 0 END AS BIT),
		@IsAVSEnabled = AVSCheck
	FROM Purchases.Card WITH(NOLOCK) WHERE InternalIssuerId = @OakCardId;
	IF(@IsCardActive = 0) /* First check if Card is active */
		BEGIN
			INSERT INTO @ResTable (IsCardActive, IsATAEnabled, IsVCCDecisioning, Id, IsAVSEnabled, EnforceCVVCheck, AllowPartialAuthorizations)
			SELECT @IsCardActive, @IsATAEnabled, @IsVCCDecisioning, 1, @IsAVSEnabled, @EnforceCVVCheck, @AllowPartialAuthorizations   
		END
	ELSE
		BEGIN
			SELECT @IdMerchant = IdMerchant FROM dbo.IncomingTransactionCode WITH(NOLOCK) WHERE IdIncomingTransactionCode = @IdIncomingTransactionCode;
			SELECT @merchantGuid = GatewayMerchantGuid, @IsATAEnabled = IsATAEnabled FROM purchases.Merchant WITH(NOLOCK) WHERE IdMerchant = @IdMerchant;
			SELECT @ParentMerchantGuid = parent.GatewayMerchantGuid, @UseParentFunding = ISNULL(parent.UseParentFunding,0) FROM purchases.merchant m
			JOIN purchases.merchant parent ON m.ParentMerchantGuid = parent.GatewayMerchantGuid
			WHERE m.GatewayMerchantGuid = @merchantguid;
			IF(@IsATAEnabled = 1) /* If ATA is enabled, go get Available Balance */
				BEGIN
					SELECT @AvailableToAuthorize = mata.AvailableToAuthorize
					FROM MerchantAvailableToAuthorize mata
					WHERE mata.GatewayMerchantGuid = IIF(@UseParentFunding = 1, @ParentMerchantGuid, @merchantguid)
				END
			/* Get VCC Decisioning details and client-level CVV validation setting */
			SELECT @IdMasterMerchant = Id, @EnforceCVVCheck = ISNULL(EnforceCVVCheck, 0), @AllowPartialAuthorizations = AllowPartialAuthorizations   
				FROM dbo.MasterMerchant WITH(NOLOCK) WHERE GatewayMerchantGuid = @merchantGuid;
			SELECT @IsVCCDecisioning = CAST(CASE WHEN IsEnabled = 1 AND WebhookEndpoint IS NOT NULL THEN 1 ELSE 0 END AS BIT), @WebhookEndpoint = WebhookEndpoint, @WebhookUserName = WebhookUserName, @WebhookPassword = WebhookPassword, @TimeoutHandling = TimeoutHandling
				FROM Bridge.MerchantVCCDecisioningSettings WITH(NOLOCK) WHERE IdMasterMerchant = @IdMasterMerchant;
			
			IF(@CardAcceptorId IS NOT NULL AND @AcquiringInstitutionIdCode IS NOT NULL)
			BEGIN
				SELECT @IsBlackListed = CASE WHEN EXISTS (
															SELECT 1
															FROM Purchases.MerchantMidBlacklist mmb
															WHERE mmb.Idmerchant = @IdMerchant
															  AND mmb.Mid = @CardAcceptorId
															  AND mmb.AcquirerId = @AcquiringInstitutionIdCode
														   ) THEN 1 ELSE 0 END;
			END
			INSERT INTO @ResTable (IsCardActive, IsATAEnabled, AvailableToAuthorize, IsVCCDecisioning, WebhookEndpoint, WebhookUserName, WebhookPassword, TimeoutHandling, VccGuid, Id, IsAVSEnabled, EnforceCVVCheck, AllowPartialAuthorizations, IsBlackListed)
				VALUES (@IsCardActive, @IsATAEnabled, @AvailableToAuthorize, @IsVCCDecisioning, @WebhookEndpoint, @WebhookUserName, @WebhookPassword, @TimeoutHandling, @VccGuid, 1, @IsAVSEnabled, @EnforceCVVCheck, @AllowPartialAuthorizations, @IsBlackListed)
		END
	RETURN
END
GO


